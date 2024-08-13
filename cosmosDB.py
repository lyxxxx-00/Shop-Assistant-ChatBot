import os
import pymongo
import requests
from pymongo import UpdateOne, DeleteMany
from models import Product, ProductList, Customer, CustomerList, SalesOrder, SalesOrderList
from dotenv import load_dotenv

load_dotenv()
CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
client = pymongo.MongoClient(CONNECTION_STRING)
# Create database to hold cosmic works data
# MongoDB will create the database if it does not exist
db = client.cosmic_works

# empty the collections
db.products.bulk_write([DeleteMany({})])
db.customers.bulk_write([DeleteMany({})])
db.sales.bulk_write([DeleteMany({})])

# Add product data to database using bulkwrite and updateOne with upsert
# Get cosmic works product data from github
product_raw_data = "https://cosmosdbcosmicworks.blob.core.windows.net/cosmic-works-small/product.json"
product_data = ProductList(items=[Product(**data) for data in requests.get(product_raw_data).json()])
db.products.bulk_write([ UpdateOne({"_id": prod.id}, {"$set": prod.model_dump(by_alias=True)}, upsert=True) for prod in product_data.items])

customer_sales_raw_data = "https://cosmosdbcosmicworks.blob.core.windows.net/cosmic-works-small/customer.json"
response = requests.get(customer_sales_raw_data)
# override decoding
response.encoding = 'utf-8-sig'
response_json = response.json()
# filter where type is customer
customers = [cust for cust in response_json if cust["type"] == "customer"]
# filter where type is salesOrder
sales_orders = [sales for sales in response_json if sales["type"] == "salesOrder"]

customer_data = CustomerList(items=[Customer(**data) for data in customers])
db.customers.bulk_write([ UpdateOne({"_id": cust.id}, {"$set": cust.model_dump(by_alias=True)}, upsert=True) for cust in customer_data.items])

sales_data = SalesOrderList(items=[SalesOrder(**data) for data in sales_orders])
db.sales.bulk_write([ UpdateOne({"_id": sale.id}, {"$set": sale.model_dump(by_alias=True)}, upsert=True) for sale in sales_data.items])

client.close()