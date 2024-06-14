import tkinter as tk

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()
        self.points = []

        self.canvas.bind("<B1-Motion>", self.add_point)

        self.save_button = tk.Button(root, text="Save", command=self.save_points)
        self.save_button.pack()

    def add_point(self, event):
        x, y = event.x, event.y
        # Przekształć współrzędne do zakresu od -100 do 100
        x_scaled = round((x - 200) / 2, 1)
        y_scaled = round(-(y - 200) / 2, 1)
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")
        self.points.append((x_scaled, y_scaled))

    def save_points(self):
        filename = "drawn_figure.txt"
        with open(filename, 'w') as file:
            for point in self.points:
                file.write(f"{point[0]},{point[1]}\n")
        print(f"Figure points saved to {filename}")

def main():
    root = tk.Tk()
    root.title("Drawing App")
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
