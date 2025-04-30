#for testing the sound file
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

try:
    # Load the sound file
    pygame.mixer.music.load("C:/Users/abhir/Downloads/alarm.wav")
    print("Sound file loaded successfully!")
    
    # Play the sound
    pygame.mixer.music.play()
    print("Playing sound...")
    
    # Keep the program running while sound plays
    time.sleep(3)
    
    # Stop the sound
    pygame.mixer.music.stop()
    print("Sound stopped")
    
except Exception as e:
    print(f"Error: {e}")

pygame.quit()
