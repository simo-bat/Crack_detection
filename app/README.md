## To use the app, first create the docker image: 

docker build -t crack_api .

## then create the image and start it:

docker run -p 5000:5000 crack_api

## Go to http://0.0.0.0:5000/ to test the API 