FROM python:3.12-slim

# Real reason for slim + minimal layers: this VM will eventually run many
# of these side by side ("tons and tons more" workers), so keeping each
# image small matters for actual disk/pull time on the box, not just style.
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scan_worker.py .

# No EXPOSE — this service makes outbound connections only (S3, Supabase,
# Overture's STAC catalog), it never listens for inbound traffic, so the
# VM's firewall never needs a port opened for it.
CMD ["python", "scan_worker.py"]
