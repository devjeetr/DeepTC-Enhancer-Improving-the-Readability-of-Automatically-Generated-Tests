
  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AnsiStyle style = AnsiStyle.NORMAL;
      String encoded = AnsiOutput.encode(style);
      Object[] data = new Object[5];
      String result = AnsiOutput.toString(data);
      assertTrue(result.equals((Object)encoded));
  }