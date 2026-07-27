# D:\Jinvexa\DataHandle\Utils\WebsiteParser.py

import asyncio
import logging
import re
from typing import Dict, Any, Optional

from bs4 import BeautifulSoup
import trafilatura
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsiteParser:
    """
    A robust website parser that uses Playwright for rendering,
    BeautifulSoup for structure extraction, and Trafilatura for clean text.
    """

    def __init__(self):
        # Patterns to detect CAPTCHAs and bot verification pages
        self.verification_patterns = [
            r"g-recaptcha",
            r"hcaptcha",
            r"turnstile",
            r"cdn-cgi",
            r"cf-challenge",
            r"verify you are human",
            r"access denied",
            r"attention required",
            r"security check"
        ]

        # Tags to remove for cleaner extraction
        self.tags_to_remove = [
            "script", "style", "noscript", "svg", "iframe", "footer", "nav"
        ]

    def parse(self, url: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for the asynchronous parsing logic.
        """
        try:
            # Try to run async
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import threading
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_parse(url))
                    return future.result()
            else:
                return asyncio.run(self._async_parse(url))
        except Exception as e:
            logger.error(f"Critical error during parsing {url}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _async_parse(self, url: str) -> Dict[str, Any]:
        """
        The core asynchronous pipeline for parsing a website.
        """
        browser = None
        try:
            async with async_playwright() as p:
                # 1. Playwright: Render the page
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0 Safari/537.36"
                )
                page = await context.new_page()

                try:
                    # Navigate and wait until network is idle
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                except Exception as e:
                    if browser:
                        await browser.close()
                    return {"status": "error", "message": f"Navigation failed: {str(e)}"}

                html_content = await page.content()
                await browser.close()
                browser = None

                # 2. Verification Detection
                content_lower = html_content.lower() if html_content else ""
                for pattern in self.verification_patterns:
                    if re.search(pattern, content_lower):
                        return {
                            "status": "verification_required",
                            "reason": "captcha",
                            "message": "Please upload a screenshot or PDF of this page."
                        }

                # 3. BeautifulSoup: Parse HTML
                soup = BeautifulSoup(html_content, "html.parser")

                # 4. Trafilatura: Extract clean main content (full_text)
                full_text = ""
                try:
                    extracted = trafilatura.extract(html_content)
                    if extracted:
                        full_text = extracted
                except Exception as e:
                    logger.warning(f"Trafilatura extraction failed: {e}")

                # If trafilatura failed, try to get text from BeautifulSoup
                if not full_text:
                    # Remove unwanted tags
                    for tag in soup.find_all(self.tags_to_remove):
                        tag.decompose()
                    full_text = soup.get_text(separator="\n", strip=True)

                # 5. Clean and extract structured elements
                # Remove unwanted tags for structured extraction
                for tag in soup.find_all(self.tags_to_remove):
                    tag.decompose()

                # Further cleaning: Remove elements with ad-related classes or IDs
                for element in soup.find_all(True):
                    try:
                        class_list = element.get("class", [])
                        id_val = element.get("id", "")
                        if any("ad-" in str(c).lower() or "sponsored" in str(c).lower() for c in class_list) or \
                           ("ad-" in id_val.lower() or "sponsored" in id_val.lower()):
                            element.decompose()
                    except:
                        continue

                # Extraction
                title = ""
                try:
                    if soup.title:
                        title = soup.title.get_text(strip=True)
                    elif soup.find("h1"):
                        title = soup.find("h1").get_text(strip=True)
                except:
                    title = url

                description = ""
                try:
                    meta_desc = soup.find("meta", attrs={"name": "description"})
                    if meta_desc:
                        description = meta_desc.get("content", "").strip()
                except:
                    pass

                headings = []
                try:
                    for h_tag in ["h1", "h2", "h3"]:
                        for h in soup.find_all(h_tag):
                            text = h.get_text(" ", strip=True)
                            if text:
                                headings.append(text)
                except:
                    pass

                paragraphs = []
                try:
                    for p in soup.find_all("p"):
                        text = p.get_text(" ", strip=True)
                        if text:
                            paragraphs.append(text)
                except:
                    pass

                links = []
                try:
                    for a in soup.find_all("a", href=True):
                        text = a.get_text(strip=True)
                        href = a["href"]
                        if text and href:
                            links.append({
                                "text": text[:100] if text else "",
                                "url": href
                            })
                except:
                    pass

                # If we have very little content, try to extract more from the page
                if len(paragraphs) < 3 and full_text:
                    # Use full_text as content
                    pass

                # Build fallback content
                if not title and not description and not paragraphs:
                    # Get all visible text
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    paragraphs = lines[:20] if lines else ["No content extracted"]

                return {
                    "status": "success",
                    "title": title or "Untitled",
                    "description": description or "",
                    "headings": headings[:20] if headings else [],
                    "paragraphs": paragraphs[:30] if paragraphs else [],
                    "links": links[:20] if links else [],
                    "images": [],
                    "tables": [],
                    "metadata": {
                        "buttons": [],
                        "forms": 0
                    },
                    "full_text": full_text[:10000] if full_text else ""  # Limit to 10000 chars
                }

        except Exception as e:
            logger.error(f"Error in _async_parse for {url}: {str(e)}")
            if browser:
                try:
                    await browser.close()
                except:
                    pass
            return {
                "status": "error",
                "message": str(e)
            }