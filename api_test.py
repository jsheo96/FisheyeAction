import requests
bulb_right = True
trigger = 'on' if bulb_right else 'off'
url = "http://192.168.0.25:8123/api/services/light/turn_{}".format(trigger)
token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI4NjBkOTdhOGY1NmM0OTY5OGVkMDZkYjg4Y2Q1ZjBmZiIsImlhdCI6MTY0NTg5NTY2MiwiZXhwIjoxOTYxMjU1NjYyfQ.L7mvKMyrRcGADZD3-SFb8-USg8HywbHy_Cq32tUy0NQ'
headers = {"Authorization": "Bearer {}".format(token), 'Content-Type': 'application/json'}
data = {"entity_id": "light.short"}
print(url)
print(headers)
print(data)
response = requests.post(url, headers=headers, json=data)
print(response.text)
