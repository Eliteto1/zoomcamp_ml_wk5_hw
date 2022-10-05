
import requests

url = "http://localhost:9696/predict"

customer_id = 'xyz-123'


client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
respose = requests.post(url, json=client).json()

print(respose)


if response['card'] == True:
    print('sending card available to %s' % customer_id)
else:
    print('Do not sending card available to %s' % customer_id)



