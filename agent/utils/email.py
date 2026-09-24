import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from agent.config import (
    SMTP_SERVER,
    SMTP_PORT,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
)

# Send an email with HTML body.
def send_email(to: str, subject: str, body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = to
    
    html_body = body.replace("\n", "<br>\n")
    html_part = MIMEText(f"<html><body>{html_body}</body></html>", "html")
    text_part = MIMEText(body, "plain")
    
    msg.attach(text_part)
    msg.attach(html_part)
    
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, to, msg.as_string())