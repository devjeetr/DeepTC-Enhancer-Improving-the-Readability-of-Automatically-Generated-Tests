@Test
public void test8()  throws Throwable  {
    CharPosition charPosition0 = new CharPosition(346, 346);
    VWordPosition vWordPosition0 = new VWordPosition(346, charPosition0);
    assertNotNull(vWordPosition0);

    boolean boolean0 = vWordPosition0.equals((WordPosition) null);
    assertEquals(false, boolean0);
    assertEquals("vertical(346;346,346)", vWordPosition0.toString());
}