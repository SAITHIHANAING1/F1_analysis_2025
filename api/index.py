import subprocess
import os
import signal

def handler(request):
    # Kill any existing Streamlit processes
    try:
        os.kill(int(open(".streamlit_pid").read()), signal.SIGTERM)
    except:
        pass

    # Launch Streamlit
    process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "3000", "--server.address", "0.0.0.0"]
    )

    with open(".streamlit_pid", "w") as f:
        f.write(str(process.pid))

    return "Streamlit app starting..."
