import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import cors from "cors";

const app = express();
app.use(cors());
const server = createServer(app);

app.use(cors({
  origin: 'http://localhost:5173', // allow frontend origin
  methods: ['GET', 'POST']
}));

const io = new Server(server, {
  cors: {
    origin: 'http://localhost:5173', // match this with your frontend
    methods: ['GET', 'POST']
  }
});

const allusers = {};

io.on("connection", (socket) => {
    console.log(`User connected: ${socket.id}`);

    socket.on("join-user", (username) => {
        allusers[username] = { username, id: socket.id };
        io.emit("joined", allusers);
    });

    socket.on("offer", ({ from, to, offer }) => {
        io.to(allusers[to]?.id).emit("offer", { from, to, offer });
    });

    socket.on("answer", ({ from, to, answer }) => {
        io.to(allusers[from]?.id).emit("answer", { from, to, answer });
    });

    socket.on("icecandidate", (candidate) => {
        socket.broadcast.emit("icecandidate", candidate);
    });

    socket.on("end-call", ({ from, to }) => {
        io.to(allusers[to]?.id).emit("end-call", { from, to });
    });

    socket.on("call-ended", ([from, to]) => {
        io.to(allusers[from]?.id).emit("call-ended", [from, to]);
        io.to(allusers[to]?.id).emit("call-ended", [from, to]);
    });
        socket.on("asl-message", ({ from, to, word }) => {
        const receiverId = allusers[to]?.id;
        if (receiverId) {
          io.to(receiverId).emit("receive-asl", { from, word });
        }
      });
      

    socket.on("disconnect", () => {
        for (const key in allusers) {
            if (allusers[key].id === socket.id) {
                delete allusers[key];
                break;
            }
        }
        io.emit("joined", allusers);
    });
});

server.listen(9000, () => {
    console.log("Server is running on http://localhost:9000");
});
