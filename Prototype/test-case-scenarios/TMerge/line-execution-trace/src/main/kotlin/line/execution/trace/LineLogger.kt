package line.execution.trace

import mu.KotlinLogging
import org.jetbrains.exposed.sql.Column
import org.jetbrains.exposed.sql.Database
import org.jetbrains.exposed.sql.SchemaUtils
import org.jetbrains.exposed.sql.Table
import org.jetbrains.exposed.sql.batchInsert
import java.io.Serializable
import org.jetbrains.exposed.sql.transactions.TransactionManager
import org.jetbrains.exposed.sql.transactions.transaction
import org.jetbrains.exposed.sql.transactions.transactionManager
import java.io.*
import java.net.Socket
import java.sql.Connection

object LineExecutionTable : Table() {
    val id: Column<Int> = integer("id").autoIncrement().primaryKey()
    val filename: Column<String> = varchar("filename", 200)
    val testClass: Column<String> = varchar("test_class", 200)
    val testCase: Column<String> = varchar("test_case", 200)
    val testLine: Column<Int> = integer("test_line")
    val line: Column<Int> = integer("line")
    val time: Column<Int> = integer("time")
}
private data class LineExecutionRecord(val filename: String, val testClass: String, val testCase: String,
                                       val testLine: Int, val line: Int, val time: Int,
                                       var commited:Boolean=false):Serializable

private data class LineExecutionRecords(val records: List<LineExecutionRecord>):Serializable
private val logger = KotlinLogging.logger {}

/**
 * Global class with static methods allowing methods to log their
 * line executions and enter/exit of methods.
 */
object LineLogger {
    private var t = 0
    private var testCase: String? = null
    private val executionRecords = mutableListOf<LineExecutionRecord>()
    private var testClass: String? = null
    private var testLine: Int = -1
    private var dbLocation: String? = null
    private var db:Database? = null

    fun initDB(dbLocation: String) {
        this.dbLocation = dbLocation
//        db = Database.connect("jdbc:sqlite:${dbLocation}", "org.sqlite.JDBC")
//        db.transactionManager.defaultIsolationLevel = Connection.TRANSACTION_SERIALIZABLE
//        transaction(db) {
//            SchemaUtils.createMissingTablesAndColumns(LineExecutionTable)
//        }

        if(!File(dbLocation).exists())
            File(dbLocation).writeText("test_class,test_case,time,test_line,filename,line\n")
    }
    fun setTestClass(testClass: String) {
        this.testClass = testClass
    }

    fun log(file: String, lineNumber: Int) {
        if (testCase == null) {
            return
        }

        t += 1
        if (this.testClass == null) {
            throw IllegalStateException("Value of testClass needs to be set before logging lines: LineLogger")
        }

//        SocketConnection.sendPacket()
//        println(file)
        val record = LineExecutionRecord(file, this.testClass.toString(), testCase.toString(), testLine,  lineNumber, t)
//        executionRecords.add(record)
       try{
           File(dbLocation.toString()).appendText(
                   "${record.testClass},${record.testCase},${record.time},${record.testLine},${record.filename},${record.line}\n")
       } catch(e: Exception) {
           println("Error during writing")
           e.printStackTrace()
       }
    }


    fun enterTestMethod(method: String) {
        logger.info("Entering $method")
        t = 0
        testCase = method
    }

    fun exitTestMethod(method: String) {
        this.flush()
        logger.info("Exiting $method")
        testCase = null
    }

    fun enterTestLine(line: Int) {
        testLine = line
    }

    private fun flush() {
        println("Writing ${executionRecords.size} records to $dbLocation")
//       try{
//           File(dbLocation.toString()).appendText(executionRecords.joinToString(separator = "") {
//               "${it.testClass},${it.testCase},${it.time},${it.testLine},${it.filename},${it.line}\n"
//           })
//       } catch(e: Exception) {
//           println("Error during writing")
//           e.printStackTrace()
//       }
        val records: LineExecutionRecords = deepCopy(LineExecutionRecords(executionRecords))
//        transaction (db){
//            LineExecutionTable.batchInsert(records.records){
//
//                    this[LineExecutionTable.filename] = "${it.filename.toString()}"
//                    this[LineExecutionTable.testClass] = "${it.testClass}"
//                    this[LineExecutionTable.testCase] = "${it.testCase}"
//                    this[LineExecutionTable.testLine] = it.testLine
//                    this[LineExecutionTable.line] = it.line
//                    this[LineExecutionTable.time] = it.time
//
//
//                it.commited = true
//            }
//            commit()
//            flushCache()
//        }
        println("finished writing")

        executionRecords.clear()
        logger.info { "Complete" }
    }
}

fun <T : Serializable> deepCopy(obj: T): T {
    val baos = ByteArrayOutputStream()
    val oos  = ObjectOutputStream(baos)
    oos.writeObject(obj)
    oos.close()
    val bais = ByteArrayInputStream(baos.toByteArray())
    val ois  = ObjectInputStream(bais)
    @Suppress("unchecked_cast")
    return ois.readObject() as T
}

