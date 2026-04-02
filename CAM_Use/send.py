#需要发送：距离、图形边长或直径、图形类别

from maix import uart, pinmap, time, sys, err
from struct import pack

pin_function = {
        "A16": "UART0_TX",
        "A17": "UART0_RX"
        # "A19": "UART1_TX"
        # "A18": "UART1_RX",

    }
device = "/dev/ttyS0"
# device = "/dev/ttyS1"

for pin, func in pin_function.items():
    err.check_raise(pinmap.set_pin_function(pin, func), f"Failed set pin{pin} function to {func}")

# Init UART
serial_dev = uart.UART(device, 115200)
serial_dev.write_str("Hello MaixPy")

# send

D=0
x=0
classs=1

bytes_content = b'\xAA\xBB\xCC\xDD'
bytes_content += pack("<i", D)    # 小端编码
bytes_content += pack("<i", x)    # 小端编码
bytes_content += pack("<i", classs)    # 小端编码
bytes_content += b'\xFF'
print(bytes_content, type(bytes_content))

serial_dev.write(bytes_content)