import sys
import time
import numpy as np
import cv2

### Lepot Functions
from pylepton.Lepton3 import Lepton3

### For Servos
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)

def capture(flip_v = False, device = "/dev/spidev0.0"):
    with Lepton3(device) as l:
        a,_   = l.capture()
        printTemp(a)

        ###### Levi's Stuff for Visualization
		# Create a Matrix.txt File, Should Comment Out Once Code is Fully Written and Tested
        Matrix = [[0 for x in range(160)] for y in range(120)]
        for i in range(120):
            for j in range(160):
               Matrix[i][j] = (((a[i,j,0]-27315)/100.00)*1.8+32)

		# Save File
        np.savetxt("chunks/Temps.txt", Matrix,fmt='% 4d');
		
		# Resize Matrix to Work With
        realchunk = [[0 for x in range(53)] for y in range(30)]
        for i in range(30): #for y axis
            for j in range(53):#for x axis
                for k in range (4):#for small y axis
                    for l in range(3):#for small x axis
                     realchunk[i][j] = realchunk[i][j] + Matrix[4 * (i) + k][3 * (j) + l]
        for i in range(30):
            for j in range(53):
                realchunk[i][j] = realchunk[i][j] /12 #average per amount of cells in chunk
		
		# Save File
        np.savetxt("realchunk.txt", realchunk,fmt='% 4d');

		# Convert to NP Array
        arr = np.array(realchunk)
        


	# Initialize Center 1x3 and Temperature 1x3 for Calculations
    centreh = np.zeros((1,3))
    temp   = np.zeros((1,3))
    
	### Iterate Over Matrix For Center Detection of Objects
	# Outer Loop
    for i in range(30):
        latch = 0 # Is Literally a Latch
		# Inner Loop
        for j in range(53): 
			# Check For Heat Signature Above 77 Degrees, Latch is for finding first High Temp in a Row
            if(arr[i][j] > 77 and latch == 0):
                temp[0][2] = j + 1
                temp[0][1] = i + 1
                temp[0][0] = temp[0][0] + 1
                latch = 1
				# Continue if there is more than one in a row
            elif(arr[i][j] > 77):
                temp[0][0] = temp[0][0] + 1 
            
        # Overwrite Centreh Value if Temp Value is Greater than Current       
        if(temp[0][0] > centreh[0][0]):     
            centreh[0][0] = temp[0][0]
            centreh[0][1] = temp[0][1]
            centreh[0][2] = temp[0][2]
        temp[0][0] = 0
        temp[0][1] = 0
        temp[0][2] = 0
    
	# Print Out Destinations of Servos
    print('Longest Value at: ', centreh[0][1],' V ',centreh[0][2], ' H, and it is ', centreh[0][0], ' Long ;p.')
    # Save File of Coordinates
	np.savetxt("chunks/centreh.txt", centreh,fmt='% 4d');
    
	# Initialize Containers For Indexing Servos and for Keeping Current Value
    current_pit = 0 # ServoP Value
    current_yaw = 0 # ServoY Value

	# Read Value From Centreh to check where Center of Thermal Body is, If 0 no Thermal Body so no Movement
    if(centreh[0][0] == 0):
        current_pit = current_pit   
        current_yaw = current_yaw 
	# If Thermal Body, then Adjust Output of Centreh to a Angle that Relates to Position
    else:
        current_pit = current_pit + centreh[0][1] - 15
        current_yaw = current_yaw + (centreh[0][0] / 2 + centreh[0][2] - 27) 
	# Output to Console the Degree Changes	
    print('Turning to ', current_pit, ' pitch, and ', current_yaw, ' yaw.') 
    
        

   if(current_pit < 135):
		#servo0 is higher servo
       kit.servo[0].angle = current_pit + 85
   else:
       kit.servo[0].angle = 50

    time.sleep(1)
    kit.servo[1].angle  = current_yaw + 85
        
	# Convert Highest Temperature to Fahrenhiet and Check if Greater than 85, if so Turn on LEDs
#   if (((a.max()-27315)/100.00)*1.8+32 > 85):
        # Hi

#           result = np.where(arr == arr.max())    
#           print 'Hottest Temp is at:'
#           y_indice,p_indice = result

#        for i in range(30000000):
#       i = i + 1
    
                
