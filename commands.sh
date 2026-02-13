#!/bin/bash
 
# Login to Azure
az login
 
# Deploy Azure Web App
az webapp up -n helloflask123 --sku F1
 
# Verify deployment
az webapp show -n helloflask123
