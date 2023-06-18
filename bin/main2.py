from utils import MetricLogger, SmoothedValue
import time

# Usage example
meter = MetricLogger()

# Update meters with sample values
meter.update(loss=0.5, accuracy=0.9, time=2.34, tizio=234)

# Access meters as attributes
# print(meter.loss)  # Print the smoothed loss value
# print(meter.accuracy)  # Print the smoothed accuracy value
# print(meter.tizio)  

# Print the complete log
# print(meter)

# Iterate over an iterable and log metrics
data_loader = [1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 5, 4, 4, 3, 3, 3, 3 ,3, 3 ,3, 3]
for obj in meter.log_every(data_loader, print_freq=2, header="Epoch: [1]"):
    # Perform operations on each object in the data_loader
    time.sleep(1)  # Simulating processing time
print("fine......................")
# Synchronize meters between processes (if needed)
meter.synchronize_between_processes()

# Add a new meter
new_meter = SmoothedValue()
meter.add_meter("new_meter", new_meter)

# Update the new meter
meter.update(new_meter=0.75)

# Print the updated log
print(meter)