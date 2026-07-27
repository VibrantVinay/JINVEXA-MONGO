# D:\Jinvexa\Prompts\conversation_prompts.py

# Prompts for the Learning Discovery Agent

CONVERSATION_PROMPTS = {
    "initial_greeting": """
    I'm your AI Learning Discovery Agent. I'll help you create a personalized learning plan.

    To get started, I need to understand:
    1. What you want to learn
    2. What you already know
    3. Your learning goals

    Let's begin!
    """,
    
    "ask_experience": """
    How would you rate your current experience with {topic}?
    (beginner, intermediate, or advanced)
    """,
    
    "ask_goal": """
    What's your primary goal for learning {topic}?
    - Understand the concepts
    - Build practical projects
    - Prepare for interviews
    - Do research in this area
    - Other (please specify)
    """,
    
    "ask_time": """
    How many hours per week can you dedicate to learning?
    This helps me create a realistic schedule.
    """,
    
    "ask_knowledge": """
    Do you have experience with {concept}?
    If yes, please describe what you know.
    """,
    
    "clarification": """
    Can you tell me more about your experience with {concept}?
    For example, have you used it in any projects?
    """,
    
    "confidence_check": """
    On a scale of 1-10, how confident are you with {concept}?
    (1 = just heard of it, 10 = can teach it to others)
    """,
    
    "gap_reveal": """
    Based on my analysis, to fully understand {topic}, you'll need to learn:
    
    {gaps}
    
    You already know:
    
    {known}
    
    I estimate this will take approximately {hours} hours.
    Would you like to proceed with this learning path?
    """,
    
    "plan_ready": """
    Great! Based on our conversation, I've created your personalized learning plan for {topic}.
    
    {plan_summary}
    
    You can view the full plan with detailed topics, projects, and resources.
    """
}

QUESTION_TEMPLATES = {
    "python": [
        "How comfortable are you with Python? (1-10)",
        "Have you used Python for any projects?",
        "Do you understand Python's async/await pattern?",
        "Are you familiar with Python decorators and context managers?",
        "Have you worked with Python libraries like NumPy or Pandas?"
    ],
    
    "statistics": [
        "Do you understand basic statistical concepts?",
        "Are you comfortable with probability theory?",
        "Do you know what a p-value represents?",
        "Have you done any statistical analysis before?"
    ],
    
    "linear_algebra": [
        "Are you familiar with matrices and matrix operations?",
        "Do you understand vectors and vector spaces?",
        "Can you explain what eigenvalues and eigenvectors are?",
        "Have you used linear algebra in any projects?"
    ],
    
    "machine_learning": [
        "Have you implemented any ML models before?",
        "Do you understand the difference between supervised and unsupervised learning?",
        "Are you familiar with gradient descent optimization?",
        "Do you know what overfitting is and how to prevent it?",
        "Have you worked with any ML frameworks like scikit-learn?"
    ],
    
    "deep_learning": [
        "Do you understand neural networks?",
        "Are you familiar with backpropagation?",
        "Do you know what activation functions are and why they're important?",
        "Have you used any deep learning frameworks like PyTorch or TensorFlow?",
        "Do you understand CNNs or RNNs?"
    ],
    
    "web_development": [
        "Are you comfortable with HTML and CSS?",
        "Do you understand JavaScript?",
        "Have you built any web applications?",
        "Are you familiar with REST APIs?",
        "Do you know what a frontend framework is?"
    ],
    
    "devops": [
        "Are you comfortable with the command line?",
        "Do you understand Linux basics?",
        "Have you used Docker before?",
        "Are you familiar with CI/CD pipelines?",
        "Do you understand cloud computing concepts?"
    ]
}