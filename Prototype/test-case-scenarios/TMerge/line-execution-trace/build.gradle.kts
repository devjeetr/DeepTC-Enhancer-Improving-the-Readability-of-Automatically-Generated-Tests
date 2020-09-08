/*
 * This file was generated by the Gradle 'init' task.
 *
 * This generated file contains a sample Kotlin application project to get you started.
 */
import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("org.jetbrains.kotlin.jvm") version "1.3.61"
    id("com.github.johnrengelman.shadow") version "4.0.4"
    application
    kotlin("plugin.serialization") version "1.3.61"
}
repositories {
    // Use jcenter for resolving dependencies.
    // You can declare any Maven/Ivy/file repository here.
    jcenter()
    mavenCentral()
}

dependencies {
    implementation(platform("org.jetbrains.kotlin:kotlin-bom"))
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
    implementation("org.ow2.asm:asm:7.3.1")
    implementation("org.ow2.asm:asm-commons:7.3.1")
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit")
    implementation("com.xenomachina:kotlin-argparser:2.0.7")
    implementation("org.jetbrains.exposed:exposed:0.17.7")
    implementation("org.jetbrains.exposed", "exposed-dao", "0.20.1")
    implementation("org.jetbrains.exposed", "exposed-jdbc", "0.20.1")
    implementation("org.xerial:sqlite-jdbc:3.21.0.1")
    implementation("io.github.microutils:kotlin-logging:1.7.7")
    implementation("ch.qos.logback:logback-classic:1.2.3")
    implementation(kotlin("stdlib", org.jetbrains.kotlin.config.KotlinCompilerVersion.VERSION)) // or "stdlib-jdk8"
//    implementation("org.jetbrains.kotlinx:kotlinx-serialization-runtime:0.14.0") // JVM dependency

}
application {
    // Define the main class for the application.
    mainClassName = "line.execution.trace.AppKt"
}

tasks {
    named<ShadowJar>("shadowJar") {
        archiveBaseName.set("shadow")
        mergeServiceFiles()
        manifest {
            attributes["Premain-Class"] = "line.execution.trace.AppKt"
            attributes["Can-Redefine-Classes"] = "true"
            attributes["Can-Retransform-Classes"] = "true"
            attributes["Can-Set-Native-Method-Prefix"] = "true"
            attributes["Implementation-Title"] = "AgentKt"
            attributes["Implementation-Version"] = rootProject.version
        }
    }
}

tasks {
    build {
        dependsOn(shadowJar)
    }
}