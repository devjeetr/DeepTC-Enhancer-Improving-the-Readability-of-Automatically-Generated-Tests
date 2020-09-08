/**
    1. Creates a new multivalued hashmap "map" and puts
       a new LinkedList "value" into it.
    2. Gets first of "value" and checks its size.
 */
@Test(timeout = 4000)
public void testGetFirst()  throws Throwable  {
    MultivaluedHashMap<List<String>, List<String>> map = new MultivaluedHashMap<List<String>, List<String>>();
    LinkedList<String> value = new LinkedList<String>();
    map.putSingle(value, value);
    List<String> result = value.getFirst(keys);
    assertEquals(0, result.size());
}