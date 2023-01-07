import requests
import time

t1 = time.time()
resp = requests.get("http://127.0.0.1:5000/get_forces?coords=1.023,4.3923,-2.234,-1.023,1.3923,-1.234,2.023,-4.3923,0.234,1.023,4.3923,-2.234,-1.023,1.3923,-1.234,2.023,-4.3923,0.234")
print(time.time() - t1)