import time
import  tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter('/root/tflite/model_nasa_bearing_CNN.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print ('---------------------------------------------')
#print (input_details)
print ('---------------------------------------------')
#print (output_details)
print ('---------------------------------------------')
# Load data
data = np.loadtxt(open('/root/tflite/bearing_test_data.csv', 'rb'), delimiter=',', skiprows=0)
for i in range(20):
	time.sleep(1)
	start = time.time()
	x = data[(0+4*i):(4+4*i)]
	x = x[np.newaxis, :, :, np.newaxis].astype('float32')
	#print ('x: ', x)

	interpreter.set_tensor(input_details[0]['index'],x)

	# Call lite model
	interpreter.invoke()
	out = interpreter.get_tensor(output_details[0]['index'])
	end = time.time()
	
	#out = out.tolist()
	#predict = out.index(max(out))
	if out >= 0.5:
		print('Bearing condition: Normal')
	if out < 0.5:
		print('Bearing condition: Fault')
	print('Inference time: ', (end - start))
	
