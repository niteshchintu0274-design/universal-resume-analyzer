from app import app


if __name__ == "__main__":
    with open("flask-server-started.txt", "w", encoding="utf-8") as marker:
        marker.write("starting\n")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    with open("flask-server-started.txt", "a", encoding="utf-8") as marker:
        marker.write("stopped\n")
