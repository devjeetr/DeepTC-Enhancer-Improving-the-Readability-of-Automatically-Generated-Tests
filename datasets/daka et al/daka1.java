@Test
public void test13()  throws Throwable  {
    ClassWriter classWriter0 = new ClassWriter((-18));
    classWriter0.visitAnnotation("", false);
    classWriter0.toByteArray();
    assertEquals(3, classWriter0.index);
}