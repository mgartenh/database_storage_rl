docker container rm cs598_database
docker run --name cs598_database -p 3306 -t -d mysql:latest
mysql --host=localhost --user=root --password="" --port=3306 < setup/create_db_schema.sql