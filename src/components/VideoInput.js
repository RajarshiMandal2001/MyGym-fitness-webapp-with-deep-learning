import React, { useState } from 'react';
import { Button } from '@mui/material';


function VideoInput() {
  const [videoUrl, setVideoUrl] = useState('');
  const [response, setResponse] = useState(null);

  const handleUpload = (event) => {
    const uploadedFile = event.target.files[0];
    const fileUrl = URL.createObjectURL(uploadedFile);
    setVideoUrl(fileUrl);
  }

  const handleDownload = async () => {
    const a = document.createElement('a');
    a.href = videoUrl;
    a.download = 'FetchedVideo.mp4';
    a.click();
    fetch('http://localhost:8000/get_video_classification').then(res => res.text()).then(data => setResponse(data)).catch(error => console.error(error));
  }

  return (
    <div>
      <input className='video-upload-btn'  type="file" onChange={handleUpload} />
      {videoUrl && (
        <div>
          <video className='video-window' src={videoUrl} controls width={500} height={500}/>
          {/* <button className='video-download-btn' onClick={handleDownload}>Download</button> */}
          <Button className='video-download-btn' onClick={handleDownload} sx={{ bgcolor: '#FF2625', color: '#fff', left:{ lg: '650px', sm: '200px' }, mt:"10px"}}>Download</Button>
        </div>
      )}
      {response ? <p>{response}</p> : <p>Result will be shown here</p>}
    </div>
  );
}

export default VideoInput;