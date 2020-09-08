@Test(timeout = 4000)
public void test029()  throws Throwable  {
    URI primaryKeyUri = MockURI.aFTPURI;
    KeycloakUriBuilder uriBuilder = KeycloakUriBuilder.fromUri(primaryKeyUri);
    String host = keycloakUriBuilder0.getHost();
    assertEquals((-1), keycloakUriBuilder0.getPort());
    assertNotNull(host);
}