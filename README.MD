



face dataset 
```
>> python ict_download.py
```



download it to project directory ```shape_predictor_68_face_landmarks.dat file```



make image meta file.


run 
```
python pyqt.py
```
and then 
open meta file directory.

now you can edit landmark detection.





# image 

60 images

15 neutral pose 
    num : 1 - 0 degree pose
    num : 6 - yaw -90 ~ 90 (per 30 degree) : this mean left and right head
    num : 4 - pitch-30 ~ 30 (per 15 degree)
    num : 4 - roll -30 ~ 30 (per 15 degree)

non-rigid pose(expression)
3 yaw angle -30, 0, 30
mouth stretch
1. smile
2. brow raise 
3. disgust 
4. squeeze left eye 
6. squeeze right eye 
7. anger
8. jaw left 
9. jaw right 
10. grin
11. chin raise
12. lip pucker
13. lip funnel
14. cheek blowing 
15. eyes closed


# predeifned_weight 
[basis_w, expr_1_w expr_2_w .... expr_n_w ]






# 2d coordinates system

all 2d landmark coordinates is followed opencv coordinates.

```
    # IMAGE
     ______________________________________________
    |  ===========================> +X Axis         |
    | |                                             |
    | |                                             |
    | |                                             |
    | |                                             |
    | |                                             |  
    | |                                             |  
    | |                                             |  
    | |                                             |  
    | V +Y Axis                                     |
    |                                               |
    |_______________________________________________|
```