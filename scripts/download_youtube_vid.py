import yt_dlp

# Download specific quality
ydl_opts = {
    'format': 'best[height<=720]',  # Max 720p
}

# Download audio only
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': '%(title)s.%(ext)s',
}

# Download to specific folder
ydl_opts = {
    'outtmpl': '../data/videos/%(title)s.%(ext)s',
}

url = 'https://www.youtube.com/watch?v=gAuJlwnUqMs'
#'https://www.youtube.com/watch?v=Xii9_oWQ7HY'
short = 'https://youtube.com/shorts/yC5nZnWXOYM'
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])