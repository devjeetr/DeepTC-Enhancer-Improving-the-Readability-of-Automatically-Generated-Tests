@Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MultivaluedHashMap<List<String>, List<String>> map = new MultivaluedHashMap<List<String>, List<String>>();
      LinkedList<String> value = new LinkedList<String>();
      queue.putSingle(value, value);
      List<String> result = queue.getFirst(keys);
      assertEquals(0, result.size());
  }