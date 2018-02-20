# Searcher

Searcher is a project to aid search and rescue operations by autonomously flying an unmanned vehicle and detect where a person is. Unmanned Aerial Vehicles (UAVs) are already in use by search and rescue teams to search for people in hard to reach locations, but all of their drones require somebody to actively fly and watch video feed. We propose a system that autonomously flies and determines if a person is in the search area using machine vision and machine learning. We plan on using a drone for this project with a camera, and possibly an infrared camera to pick up heat signatures. Basic success would require developing an algorithm which would automatically detect if there is a person in a aerial image with some amount of precision about 75% accuracy, and integrate the algorithm into a drone in near real-time. Near real-time is defined as the processing of frames coming once every few seconds, which would still be possible and useful if the UAV is flying fairly slowly. Stretch goals would be to have multiple autonomous UAVs working together to search for a person with very high accuracy, ~95% with few false positives. Also, adding a feature that would allow the human operators to verify if a person identified by the system is the person they are looking for. The project will involve testing with a drone, which will require one.

Note:
Let's try to keep master as completly working code.
For development branch off your own branch and get it working
before merging the master with that branch.
