## To Run
1. Build docker file to create docker image
   `docker build -t fastapidemo:v1 .`
2. Run docker image 
   `docker run --rm -p 80:80 fastapidemo:v1`
3. Go to http://0.0.0.0:80/docs to test API