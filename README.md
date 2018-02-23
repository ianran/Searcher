# Searcher

Searcher is a project to aid search and rescue operations by autonomously flying an unmanned vehicle and detect where a person is. Unmanned Aerial Vehicles (UAVs) are already in use by search and rescue teams to search for people in hard to reach locations, but all of their drones require somebody to actively fly and watch video feed. We propose a system that autonomously flies and determines if a person is in the search area using machine vision and machine learning. We plan on using a drone for this project with a camera, and possibly an infrared camera to pick up heat signatures. Basic success would require developing an algorithm which would automatically detect if there is a person in a aerial image with some amount of precision about 75% accuracy, and integrate the algorithm into a drone in near real-time. Near real-time is defined as the processing of frames coming once every few seconds, which would still be possible and useful if the UAV is flying fairly slowly. Stretch goals would be to have multiple autonomous UAVs working together to search for a person with very high accuracy, ~95% with few false positives. Also, adding a feature that would allow the human operators to verify if a person identified by the system is the person they are looking for. The project will involve testing with a drone, which will require one.
  
Note:  
Let's try to keep master as completely working code.  
For development branch off your own branch and get it working  
before merging the master with that branch.  


Git cheat sheet:  
[] = arguments  
// = comments on commands  

# List of useful commands

git status                          // Tells you changed files and staged files.  
                                    // Staged files in green, unstaged files in red.  
git add [files to add to staging]   // moves files to add to staging to staged  
git add --all                       // moves all changed files to staged  
git commit -m "[commit message]"    // Commits all staged files with commit message.  
git commit                          // Commits, and starts default text editor for commit message  
                                        // Note: Default editor is often vi which can be closed with esc :x  
git push                            // Pushes local commits to server  
git pull                            // Pulls data from server, also merges at same time.  

git fetch                           // Generic pulls data from server, does not merge.  
git log                             // Shows list of recent commits.  
git checkout [branch to checkout]   // Changes your repo to a different branch, note: can go to any arbitrary commit  
git branch -v                       // Shows list of branches  
git branch [branch name]            // Creates a new branch at the current branch.  
git merge [branch name]             // Merges your current branch with [branch name]  
git stash                           // Doesn't commit, but cleans working branch. (ONLY use if you want to get rid of current  
                                        unstaged data, after checking out another commit) (I don't know exactly what it does to be honest)  


# Generic commit:

git add [files to commit]  
git commit -m "[commit message]"  
git pull  
git push  


# Starting new development branch:

git checkout master  
git branch [dev branch name]  
git checkout [dev branch name]  
...  // various commits until code is working, may need to merge with somebody else's branch as well.  
git merge master  
... // make sure code is still working with latest master branch  
git status // MAKE SURE that working branch is clean (don't need to commit)  
git checkout master  
git merge [dev branch name]  
    // may need to commit again.  
git branch -d [dev branch name]  
