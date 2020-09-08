/**
 * 1. Encodes a an AnsiStyle and stores the result
 *    in "string0".
 * 2. Creates a new object array, converts it to
 *    string using AnsiOutput, and checks if it
 *    is equal to "string0"
 */
@Test(timeout = 4000)
public void test03()  throws Throwable  {
    AnsiStyle ansiStyle0 = AnsiStyle.NORMAL;
    String string0 = AnsiOutput.encode(ansiStyle0);
    Object[] objectArray0 = new Object[5];
    String string1 = AnsiOutput.toString(objectArray0);
    assertTrue(string1.equals((Object)string0));
}