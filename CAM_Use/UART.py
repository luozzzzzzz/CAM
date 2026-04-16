from maix import uart, pinmap, time, sys, err, app
import threading
# ports = uart.list_devices() # 列出所有串口
pin_function = {
    "A18": "UART1_RX",
    "A19": "UART1_TX"
}
device = "/dev/ttyS1"

for pin, func in pin_function.items():
    err.check_raise(pinmap.set_pin_function(pin, func), f"Failed set pin{pin} function to {func}")

# Init UART
serial_dev = uart.UART(device, 115200)
serial_dev.write_str("Hello MaixPy")
"""
def task1():
    while 1:
        serial_dev.write_str("good")
        time.sleep_ms(5)
def task2():
    while 1:
        data=serial_dev.read()
        if data:
            print(data)
        time.sleep_ms(5)
send=threading.Thread(target=task1)
receive=threading.Thread(target=task2)

send.start()
receive.start()

send.join()
receive.join()
print("no")
"""
while not app.need_exit():
    #print("yes")
    serial_dev.write_str("good")
    time.sleep_ms(5)