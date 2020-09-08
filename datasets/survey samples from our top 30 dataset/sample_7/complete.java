
/**
 * 1. Encodes a an AnsiStyle and stores the result
 *    in "encoded".
 * 2. Creates a new object array, converts it to
 *    string using AnsiOutput, and checks if it
 *    is equal to "encoded"
 */
  @Test(timeout = 4000)
  public void shouldSerializeDateRange()  throws Throwable  {
      AnsiStyle style = AnsiStyle.NORMAL;
      String encoded = AnsiOutput.encode(style);
      Object[] data = new Object[5];
      String result = AnsiOutput.toString(data);
      assertTrue(result.equals((Object)encoded));
  }