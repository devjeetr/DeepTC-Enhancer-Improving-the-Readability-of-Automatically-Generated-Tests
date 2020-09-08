/**
    1. Creates a new multivalued hashmap "multivaluedHashMap0" and puts
       a new LinkedList "linkedList0" into it.
    2. Gets first of "multivaluedHashMap0" and checks its size.
 */
@Test(timeout = 4000)
public void test03()  throws Throwable  {
    MultivaluedHashMap<List<String>, List<String>> multivaluedHashMap0 = new MultivaluedHashMap<List<String>, List<String>>();
    LinkedList<String> linkedList0 = new LinkedList<String>();
    multivaluedHashMap0.putSingle(linkedList0, linkedList0);
    List<String> list0 = multivaluedHashMap0.getFirst(linkedList0);
    assertEquals(0, list0.size());
}