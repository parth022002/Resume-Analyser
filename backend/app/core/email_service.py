import logging
import urllib.request
import json
import os
import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)

# Verified Resend sandbox owner email
RESEND_SANDBOX_OWNER = "parthahuja9.pa@gmail.com"

def send_email_via_resend(to_email: str, subject: str, html_content: str):
    """
    Sends email notification using Resend REST API with automatic User-Agent header
    and fallback to owner email for Resend sandbox restrictions.
    """
    api_key = settings.RESEND_API_KEY or os.getenv("RESEND_API_KEY", "")
    if not api_key:
        logger.info(f"[SIMULATED EMAIL] To: {to_email} | Subject: {subject}")
        return {"status": "simulated", "message": "Email logged to console"}

    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Attempt 1: Try sending to candidate email + owner email
    recipients = [to_email]
    if to_email != RESEND_SANDBOX_OWNER and not to_email.endswith("@talentforge.ai"):
        recipients.append(RESEND_SANDBOX_OWNER)

    payload = {
        "from": "TalentForge Security <onboarding@resend.dev>",
        "to": recipients,
        "subject": subject,
        "html": html_content
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode()
            logger.info(f"Resend Email Sent Successfully to {recipients}: {body}")
            return {"status": "sent", "resend_response": body}
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'read'):
            try:
                error_msg = e.read().decode()
            except Exception:
                pass
        
        logger.warning(f"Resend API initial attempt failed for {recipients}: {error_msg}")

        # Attempt 2: Fallback to verified owner email (bypasses Resend 403 sandbox domain restriction)
        fallback_payload = {
            "from": "TalentForge Security <onboarding@resend.dev>",
            "to": [RESEND_SANDBOX_OWNER],
            "subject": f"[Candidate: {to_email}] {subject}",
            "html": f"<div style='background:#fef3c7;padding:10px;margin-bottom:15px;border-radius:8px;font-size:12px;color:#92400e;'><strong>Sandbox Notice:</strong> Email originally intended for candidate: <b>{to_email}</b></div>" + html_content
        }

        try:
            fallback_data = json.dumps(fallback_payload).encode("utf-8")
            req = urllib.request.Request(url, data=fallback_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read().decode()
                logger.info(f"Resend Email Sent Successfully via Owner Fallback ({RESEND_SANDBOX_OWNER}): {body}")
                return {"status": "sent_fallback", "recipient": RESEND_SANDBOX_OWNER, "resend_response": body}
        except Exception as fallback_err:
            logger.error(f"Resend API Owner Fallback failed: {fallback_err}")
            return {"status": "failed", "detail": str(fallback_err)}

def send_registration_email(full_name: str, email: str, password: str):
    """
    Sends registration confirmation email containing username & password.
    """
    subject = f"Welcome to TalentForge, {full_name}! 🚀 Your Registration Credentials"
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 16px; background-color: #ffffff;">
        <h2 style="color: #4f46e5; margin-bottom: 5px;">Welcome to TalentForge! 🚀</h2>
        <p style="color: #64748b; font-size: 14px;">Your student candidate account has been created successfully.</p>
        <hr style="border: none; border-top: 1px solid #f1f5f9; margin: 20px 0;" />
        <div style="background-color: #f8fafc; padding: 18px; border-radius: 12px; border: 1px solid #e2e8f0;">
            <h4 style="margin-top: 0; color: #1e293b; font-size: 15px;">Your Student Login Credentials</h4>
            <p style="margin: 8px 0; font-size: 14px;"><strong>Full Name:</strong> {full_name}</p>
            <p style="margin: 8px 0; font-size: 14px;"><strong>Registered Email / Username:</strong> <span style="color: #4f46e5; font-weight: bold;">{email}</span></p>
            <p style="margin: 8px 0; font-size: 14px;"><strong>Account Password:</strong> <code style="background: #e2e8f0; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-weight: bold;">{password}</code></p>
            <p style="margin: 8px 0; font-size: 14px;"><strong>Account Plan:</strong> Free Student Account</p>
        </div>
        <p style="margin-top: 20px; font-size: 13px; color: #475569; line-height: 1.5;">
            Log in to your account at <a href="http://localhost:5173" style="color: #4f46e5; font-weight: bold;">TalentForge Platform</a> to complete compulsory education details, build ATS LaTeX resumes, and view live high-match job postings.
        </p>
        <p style="font-size: 12px; color: #94a3b8; margin-top: 30px;">TalentForge AI Candidate Knowledge Graph & Sourcing System</p>
    </div>
    """
    return send_email_via_resend(email, subject, html_content)

def send_login_notification_email(full_name: str, email: str):
    """
    Sends security email notification whenever user logs in.
    """
    now_str = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    subject = f"Security Alert 🔐 Successful Login to TalentForge Account"
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 16px; background-color: #ffffff;">
        <h3 style="color: #059669; margin-bottom: 5px;">Security Alert: Successful Login 🔐</h3>
        <p style="color: #475569; font-size: 14px;">Hi {full_name},</p>
        <p style="color: #64748b; font-size: 13px;">We detected a successful login to your TalentForge student account.</p>
        <div style="background-color: #f0fdf4; padding: 16px; border-radius: 12px; border: 1px solid #bbf7d0; margin: 15px 0;">
            <p style="margin: 4px 0; font-size: 13px;"><strong>Account Email:</strong> {email}</p>
            <p style="margin: 4px 0; font-size: 13px;"><strong>Timestamp:</strong> {now_str}</p>
            <p style="margin: 4px 0; font-size: 13px;"><strong>Location / Session:</strong> Web Candidate Portal (Bengaluru, KA)</p>
        </div>
        <p style="font-size: 12px; color: #64748b;">
            If this was you, no further action is required. If you did not initiate this login, please reset your password immediately.
        </p>
    </div>
    """
    return send_email_via_resend(email, subject, html_content)
