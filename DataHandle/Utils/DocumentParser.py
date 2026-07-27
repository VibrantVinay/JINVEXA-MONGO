import os
import tempfile

import fitz
import ollama
from docx import Document

from Config.Config import Config


class DocumentParser:

    def __init__(self):

        self.config = Config()
        self.model = self.config.get_model()

    def _ocr_page(self, page):

        pix = page.get_pixmap(dpi=300)

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as f:

            img = f.name

        pix.save(img)

        try:

            r = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract every piece of text from this document exactly as written.\n"
                            "Do not summarize.\n"
                            "Do not correct grammar.\n"
                            "Return only the extracted text."
                        ),
                        "images": [img]
                    }
                ]
            )

            return r["message"]["content"].strip()

        finally:

            if os.path.exists(img):
                os.remove(img)

    def _pdf(self, path):

        doc = fitz.open(path)

        pages = []
        full = []

        for i, page in enumerate(doc):

            text = page.get_text().strip()

            # If no selectable text exists, use the model as OCR.
            if not text:
                text = self._ocr_page(page)

            pages.append({
                "page": i + 1,
                "text": text
            })

            full.append(text)

        doc.close()

        return {
            "file_name": os.path.basename(path),
            "file_type": "pdf",
            "page_count": len(pages),
            "full_text": "\n".join(full),
            "pages": pages
        }

    def _docx(self, path):

        doc = Document(path)

        paragraphs = []
        full = []

        for i, para in enumerate(doc.paragraphs):

            text = para.text.strip()

            if text:

                paragraphs.append({
                    "paragraph": i + 1,
                    "text": text
                })

                full.append(text)

        return {
            "file_name": os.path.basename(path),
            "file_type": "docx",
            "paragraph_count": len(paragraphs),
            "full_text": "\n".join(full),
            "paragraphs": paragraphs
        }

    def _text(self, path):

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "file_name": os.path.basename(path),
            "file_type": "text",
            "full_text": text
        }

    def parse(self, path):

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            return self._pdf(path)

        elif ext == ".docx":
            return self._docx(path)

        elif ext in [".txt", ".md", ".py"]:
            return self._text(path)

        raise Exception(f"Unsupported file type: {ext}")