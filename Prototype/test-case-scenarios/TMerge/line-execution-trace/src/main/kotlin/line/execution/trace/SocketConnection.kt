package line.execution.trace

import java.io.PrintWriter
import java.net.Socket

//fun createSocketConnection():Socket{
//
//}

object SocketConnection{
    fun sendPacket() {
        try{
            val socket = Socket("127.0.0.1", 5900)
            val out = PrintWriter(socket.getOutputStream(), true)

            out.println("sending a packet with love")

            out.close()
            socket.close()
        } catch(e: Exception) {
            e.printStackTrace()
        }
    }
}