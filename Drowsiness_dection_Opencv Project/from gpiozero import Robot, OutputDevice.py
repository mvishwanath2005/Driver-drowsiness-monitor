from gpiozero import Robot, OutputDevice
from time import sleep

# 1. Setup the Walking Motors (L298N)
# defined as (Forward_Pin, Backward_Pin)
bot = Robot(left=(17, 27), right=(22, 23))

# 2. Setup the Tools (Relays)
cutter = OutputDevice(5)  # Cutter on GPIO 5
sprayer = OutputDevice(6) # Sprayer on GPIO 6

print("System Active. Commands: Forward, Cut, Spray")

# Example Sequence
try:
    # Walk Forward at 50% speed
    bot.forward(0.5)
    sleep(2)
    bot.stop()

    # Turn on Cutter
    print("Cutting weeds...")
    cutter.on()
    sleep(3)
    cutter.off()

    # Turn on Sprayer
    print("Spraying...")
    sprayer.on()
    sleep(2)
    sprayer.off()

except KeyboardInterrupt:
    # Safety: Turn everything off if you press Ctrl+C
    bot.stop()
    cutter.off()
    sprayer.off()