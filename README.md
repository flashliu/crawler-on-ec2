# How to run in locally?

## 1. Install packages

```bash
pip install -r requirements.txt
```

## 2. Start the server

```bash
fastapi dev main.py
```

# How to deploy to server?

## 1. Build image

```bash
docker buildx build .
```

## 2. Save image to `.tar` file

```bash
docker save -o crawler.tar <image id or name>
```

## 3. Upload the file to server

```bash
scp -i ai-crawler.pem crawler.tar ubuntu@ec2-18-205-235-203.compute-1.amazonaws.com:~
# with proxy
scp -o "ProxyCommand=nc -x 127.0.0.1:7890 %h %p" -i ai-crawler.pem crawler.tar ubuntu@ec2-18-205-235-203.compute-1.amazonaws.com:~
```

## 4. Load the `.tar` file on server
```bash
sudo docker load -i crawler.tar
```

## 5. Start the image 
```bash
sudo docker run -d -p 80:80 <image id or name>
```

### Tips: You must download the model file to the common folder, using those links below
- https://huggingface.co/google/owlvit-base-patch32
- https://huggingface.co/facebook/bart-large-mnli