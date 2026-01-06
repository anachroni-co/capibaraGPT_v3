# Fix 500 Error Guide

## Step 1: Authenticate with Google Cloud

```cmd
gcloud auth login
```

## Step 2: Run the Repair Script

```cmd
fine-tuning/scripts/fix_500_error.bat
```

## Step 3: If Script Doesn't Work, Run Manually

### Connect to VM:
```cmd
gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001"
```

### On the VM, run these commands:

```bash
# 1. Check running processes
ps aux | grep python | grep -v grep

# 2. Check ports in use
sudo netstat -tuln | grep -E ":(5001|8080)"

# 3. Verify model responds
curl http://localhost:8080/health

# 4. Stop integrated server if running
pkill -f capibara6_integrated_server

# 5. Go to backend directory
cd ~/capibara6/backend

# 6. Verify file exists
ls -la capibara6_integrated_server.py

# 7. Start the server
nohup python3 capibara6_integrated_server.py > ../logs/server.log 2>&1 &

# 8. Wait 3 seconds and verify
sleep 3
curl http://localhost:5001/health

# 9. Check logs for errors
tail -30 ../logs/server.log
```

## Step 4: Verify in Browser

After restarting the server, test in browser:
- Open https://www.capibara6.com
- Send a test message
- If error persists, check logs

## Useful Diagnostic Commands

```bash
# View logs in real-time
tail -f ~/capibara6/logs/server.log

# View recent errors
tail -50 ~/capibara6/logs/errors.log

# Check Python processes
ps aux | grep python

# Verify server connection to model
curl -v http://localhost:8080/completion -X POST -H "Content-Type: application/json" -d '{"prompt":"test","n_predict":10}'

# Restart entire system
pkill -f python
sleep 2
cd ~/capibara6/backend
nohup python3 capibara6_integrated_server.py > ../logs/server.log 2>&1 &
```

## Common Problems and Solutions

### 1. Error: "Address already in use"
```bash
# Free port 5001
sudo fuser -k 5001/tcp
# Or
sudo kill -9 $(lsof -t -i:5001)
```

### 2. Error: "Connection refused" on port 8080
```bash
# Check if model is running
ps aux | grep llama-server
# If not, start it according to your configuration
```

### 3. Error: "Module not found"
```bash
# Install dependencies
cd ~/capibara6/backend
pip3 install -r requirements.txt
```

### 4. Error: "Permission denied"
```bash
# Check permissions
chmod +x capibara6_integrated_server.py
```

## Final Verification

After restart, verify everything is working:

```bash
# 1. Integrated server responds
curl http://localhost:5001/health

# 2. Model responds
curl http://localhost:8080/health

# 3. /api/chat endpoint works
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","conversation":[]}'
```

If all these commands work, the backend should be operational.
