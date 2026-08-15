import re

def redact_pii(text: str) -> str:
    """
    Tier 1 PII Redaction Filter:
    Redacts email addresses, phone numbers, and sensitive candidate personal information
    before sending prompt payloads to external LLMs.
    """
    if not text:
        return ""
    
    # Redact email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
    
    # Redact phone numbers (various formats)
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    text = re.sub(phone_pattern, '[REDACTED_PHONE]', text)
    
    return text
