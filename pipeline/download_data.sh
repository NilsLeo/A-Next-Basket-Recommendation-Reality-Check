#!/bin/bash

source ../.env

cd ..

# Clear existing datasets
rm -rf DataSource/
mkdir -p DataSource/dunnhumby_50k
mkdir -p DataSource/instcart

echo "Downloading datasets..."

# Tafeng
curl "$TAFENG_URL" -o DataSource/ta_feng_all_months_merged.csv.zip
unzip DataSource/ta_feng_all_months_merged.csv.zip -d DataSource/
rm DataSource/ta_feng_all_months_merged.csv.zip

# Dunnhumby  
curl "$DUNNHUMBY_URL" -o DataSource/dunnhumby.zip
unzip DataSource/dunnhumby.zip -d DataSource/
mv DataSource/'dunnhumby_Let'\''s-Get-Sort-of-Real-(Sample-50K-customers)'/* DataSource/dunnhumby_50k/
rmdir DataSource/'dunnhumby_Let'\''s-Get-Sort-of-Real-(Sample-50K-customers)'
rm DataSource/dunnhumby.zip

# Instacart
curl "$INSTACART_ORDERS_URL" -o DataSource/instcart/orders.csv.zip
curl "$INSTACART_PRODUCTS_TRAIN_URL" -o DataSource/instcart/order_products__train.csv.zip
curl "$INSTACART_PRODUCTS_PRIOR_URL" -o DataSource/instcart/order_products__prior.csv.zip

# Unzip Instacart files
unzip DataSource/instcart/orders.csv.zip -d DataSource/instcart/
unzip DataSource/instcart/order_products__train.csv.zip -d DataSource/instcart/
unzip DataSource/instcart/order_products__prior.csv.zip -d DataSource/instcart/

# Clean up zip files
rm DataSource/instcart/orders.csv.zip
rm DataSource/instcart/order_products__train.csv.zip
rm DataSource/instcart/order_products__prior.csv.zip

echo "Download completed."