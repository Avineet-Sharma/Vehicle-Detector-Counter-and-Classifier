# Vehicle Counting and Classification

This program automates the classification of the vehicles on a busy moving road

This also has a capability to record the no.of vehicles that are classified in the realtime and store the data to a csv file.

The data is helpful in decision making to government agencies like traffic and road safety dept. and Infrastructure development and planning.



Use Case :

- Automate the monitoring of traffic flow
- realtime surveillance system speard all over the many metropolitian cities.
- Security Enforcement to plan a secure route to transfer or mobilize VIP or criminals
- Automate parking systems and provide efficient parking solutions



Future scope:

- This system can be integrated with the ANPR(Automated Number Plate Recognition) system to track down the vehicle and analyze any suspicious activity.

- The combined system can also be used for collecting of efficient road utilization tax by the vehicles.



The project has 2 use cases,

    1. Using an Image to detect the no. of vehciles present in the picture

    2. Live Video(Recorded or from Camera) to analyze, capture the vehicles Count



## Step to Execute

1. Fork the Repository
2. Clone the repository to your local system
3. Python version 3.5 and above
4. Install the opencv-python dependancy using `pip install opencv-python`
5. Download the required [yolo-v3](https://drive.google.com/drive/folders/1XHBuwzZARn-8xTPNW8PRpo33A0iRaSLK?usp=sharing) files to the working directory 
6. Navigate to the `vehicle_count.py` file, you will find the implementation of the code
   - main function has 2 function calls
     - realTime()
     - from_static_image(img-file)
   realTime() demonstrates the video analysis
   from_static_image() demonstrate single picture analysis



### Inspiration

Blog - techvidvan - vehicle detection 

Videos - Youtube
