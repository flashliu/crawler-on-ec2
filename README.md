# How to deploy?

## 1. Build image

```bash
docker build .
```

## 2. Save image to `.tar` file

```bash
docker save -o crawler.tar <image id or image name>
```

## 3. Upload the file to server

```bash
scp -i ai-crawler.pem crawler.tar ubuntu@ec2-18-205-235-203.compute-1.amazonaws.com:~
```