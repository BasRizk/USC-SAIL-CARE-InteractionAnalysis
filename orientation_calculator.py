import os
import cv2
import numpy as np

def draw(img_raw, dets, save_file_prefix, vis_thres=0.6):    
    # cv2.imwrite(f'{save_file_prefix}_org.jpg', img_raw)

    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        
        eye_l_pt, eye_r_pt = (b[5], b[6]), (b[7], b[8])
        nose_pt = (b[9], b[10])
        mouth_l_pt, mouth_r_pt = (b[11], b[12]), (b[13], b[14])

        cv2.circle(img_raw, eye_l_pt, 1, (0, 0, 255), 4)
        cv2.circle(img_raw, eye_r_pt, 1, (0, 255, 255), 4)
        cv2.circle(img_raw, nose_pt, 1, (255, 0, 255), 4)
        cv2.circle(img_raw, mouth_l_pt, 1, (0, 255, 0), 4)
        cv2.circle(img_raw, mouth_r_pt, 1, (255, 0, 0), 4)

        line_color = (255, 255, 255)
        cv2.line(img_raw, eye_l_pt, eye_r_pt, line_color, 1) 
        btw_eyes_pt = center(eye_l_pt, eye_r_pt)
        cv2.circle(img_raw, btw_eyes_pt, 1, (0, 155, 155), 2)

        cv2.line(img_raw, mouth_l_pt, mouth_r_pt, line_color, 1)
        btw_mouth_pt = center(mouth_l_pt, mouth_r_pt)
        cv2.circle(img_raw, btw_mouth_pt, 1, (0, 155, 155), 2)

        cv2.line(img_raw, btw_eyes_pt, nose_pt, line_color, 1)
        cv2.line(img_raw, btw_mouth_pt, nose_pt, line_color, 1)
        cv2.line(img_raw, btw_mouth_pt, btw_eyes_pt, line_color, 1)
        # cv2.line(img_raw, eye_l_pt, mouth_r_pt, line_color, 1)
        # cv2.line(img_raw, eye_r_pt, mouth_l_pt, line_color, 1)
        
        # below_nose_pt = get_intersection_pt(
        #     [eye_l_pt, mouth_r_pt],
        #     [eye_r_pt, mouth_l_pt])
        
        # print(below_nose_pt)
        # print(eye_l_pt, eye_r_pt)
        # print(mouth_l_pt, mouth_r_pt)
        below_nose_pt = center(btw_eyes_pt, btw_mouth_pt)
        cv2.line(img_raw, below_nose_pt, nose_pt, line_color, 1)



        looking_arrow = translation(
            [below_nose_pt, nose_pt],
            y = distance(btw_eyes_pt, btw_mouth_pt)//2,
            y_dir = -1
        )

        cv2.arrowedLine(
            img_raw, looking_arrow[0], looking_arrow[1],
            line_color, 1 
        )

        # DEBUG
        cv2.line(
            img_raw, btw_eyes_pt, looking_arrow[0],
            line_color, 1 
        )            

    # save images
    cv2.imwrite(f"{save_file_prefix}_proc.jpg", img_raw)


def center(pt1, pt2):
    return (pt2[0] - pt1[0])//2 + pt1[0],\
            (pt2[1] - pt1[1])//2 + pt1[1]

def slope(pt1, pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0]),

def translation(ptArr, x=0, y=0, x_dir=1, y_dir=-1):
    transArr = []
    for i, j in ptArr:
        transArr.append((i+x*x_dir, j+y*y_dir))
    return transArr

def distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2)).astype(np.int32)

# def get_intersection_pt(line_1, line_2):
#     A = np.array(line_1)
#     print('line1', A)
#     B = np.array(line_2)
#     print('line2', B)
#     return tuple(np.linalg.solve(np.array([A[1]-A[0], B[0]-B[1]]).T, B[0]-A[0]).astype(np.int32))