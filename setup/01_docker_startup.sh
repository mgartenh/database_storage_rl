cd setup
docker compose down
docker compose up --detach

sleep 20

mysql --host=127.0.0.1 --user=root --password="password" --port=3307 < create_db_schema.sql

cd ..