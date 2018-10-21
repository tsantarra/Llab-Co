import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# basic to/from info
fromaddr = 'trevor.santarra@gmail.com'
recipients = ['trevor.santarra@gmail.com']

# Set up the message
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = ', '.join(recipients)
msg['Subject'] = 'SUBJECT OF THE MAIL'
msg.attach(MIMEText('HAI FROM PYTHON', 'plain'))

# The actual sending part
with smtplib.SMTP('smtp.gmail.com', 587) as server:
    server.starttls()
    server.login(fromaddr, 'Joseph88@')
    server.sendmail(fromaddr, recipients, msg.as_string())


