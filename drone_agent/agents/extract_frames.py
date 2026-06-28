import cv2                  #reads video files
import os                   #creates folder build file paths    
import json                 #save metadata as .json file


def extract_frames(video_path:str ,output_folder:str,every_nth_frame : int=6) -> list[dict]:
    """
        Read a video file and save every nth frame

        Args:
        video path : input video path
        output_folder : where to save the output frames.
        every_n_second : how often to grab a 

        Returns:
        List of dicts, each describing one saved frames

    """

     
    ########################## Open the video file #############################
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video:{video_path}")



    ############################ Read video properties #############################
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds=total_frames/ fps

    print("Video loaded")
    print(f"fps :{fps}")
    print(f"total frames :{total_frames}")
    print(f"duration_seconds :{duration_seconds}")


    ##################### create output folder for extracted frames ####################
    os.makedirs(output_folder, exist_ok=True)
    

    frame_interval= every_nth_frame
    
    ######################### Loop through the video ###################################
    frame_metadata =[]
    frame_index=0
    saved_count=0

    while True:
        ## cap.read()
        ## return:
        ## success:True if a frame was read and false if video ended
        ## frame: the actual image as numpy array( height*weidth*3)

        success, frame = cap.read()

        if not success:
            break
        
        ############## save the frame only if it's right time interval  ####################
        if frame_index % frame_interval == 0:
            ############ calculate the real timestamp ##########
            timestamp_sec = frame_index / fps

            ########### Build the filename like: frame_0042_at_84.0s.jpg ################
            filename = f'frame_{saved_count:04d}_at_{timestamp_sec:.1f}s.jpg'
            filepath = os.path.join(output_folder,filename)


            ############## save the images to disk ################
            cv2.imwrite(filepath, frame)

            ############### store the metadata about this frame #######################
            frame_metadata.append({
                "frame_id": f'frame_{saved_count:04d}',
                "filename":filename,
                "filepath":filepath,
                "timestamp_sec":round(timestamp_sec,2),
                "frame_index":frame_index

            })

            saved_count +=1
            print(f" saved : {filename}")

        frame_index +=1
    print(f"saved count: {saved_count}")
    ############# Release the video file ##################
    cap.release()

    ########## save the metadata as json #############
    metadata_path = os.path.join(output_folder,'frames_metadata.json')
    with open(metadata_path,'w') as f:
        json.dump(frame_metadata,f,indent=2)

    return frame_metadata

