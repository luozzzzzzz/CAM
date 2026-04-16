from maix import pinmap, uart, app, time

serial_dev = uart.UART("/dev/ttyS1", 115200)
serial_dev.write_str("Hello MaixPy")

while not app.need_exit():
    serial_dev.write_str("Hello MaixPy")
    #time.sleep_ms(1)
