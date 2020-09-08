
/**
    1. Creates a new KeyCloakUriBuilder using aFTPURI of MockURI,
       and checks if its port is -1 and host is not null.
  */
@Test(timeout = 4000)
public void testUri()  throws Throwable  {
    URI primaryKeyUri = MockURI.aFTPURI;
    KeycloakUriBuilder uriBuilder = KeycloakUriBuilder.fromUri(primaryKeyUri);
    String host = uriBuilder.getHost();
    assertEquals((-1), uriBuilder.getPort());
    assertNotNull(host);
}