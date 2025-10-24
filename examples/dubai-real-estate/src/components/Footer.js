import React from 'react';
import { motion } from 'framer-motion';
import { FaBuilding, FaPhone, FaEnvelope, FaMapMarkerAlt, FaFacebook, FaInstagram, FaLinkedin, FaWhatsapp, FaTelegram } from 'react-icons/fa';
import './Footer.css';

const Footer = () => {
  const footerLinks = {
    company: [
      { name: 'О компании', href: '#about' },
      { name: 'Наша команда', href: '#about' },
      { name: 'Карьера', href: '#' },
      { name: 'Новости', href: '#' }
    ],
    services: [
      { name: 'Продажа недвижимости', href: '#services' },
      { name: 'Аренда', href: '#services' },
      { name: 'Инвестиции', href: '#services' },
      { name: 'Консультации', href: '#contact' }
    ],
    properties: [
      { name: 'Виллы', href: '#properties' },
      { name: 'Апартаменты', href: '#properties' },
      { name: 'Пентхаусы', href: '#properties' },
      { name: 'Коммерческая недвижимость', href: '#properties' }
    ]
  };

  const socialLinks = [
    { icon: FaFacebook, href: '#', name: 'Facebook' },
    { icon: FaInstagram, href: '#', name: 'Instagram' },
    { icon: FaLinkedin, href: '#', name: 'LinkedIn' },
    { icon: FaWhatsapp, href: '#', name: 'WhatsApp' },
    { icon: FaTelegram, href: '#', name: 'Telegram' }
  ];

  const scrollToSection = (href) => {
    if (href.startsWith('#')) {
      const element = document.querySelector(href);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <motion.div
            className="footer-section footer-brand"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <div className="footer-logo">
              <FaBuilding className="logo-icon" />
              <span>Dubai Elite Properties</span>
            </div>
            <p className="footer-description">
              Ваш надежный партнер в мире премиальной недвижимости Дубая. 
              Более 15 лет опыта работы на рынке недвижимости ОАЭ.
            </p>
            <div className="footer-contact">
              <div className="contact-item">
                <FaPhone />
                <span>+971 4 123 4567</span>
              </div>
              <div className="contact-item">
                <FaEnvelope />
                <span>info@dubaieliteproperties.ae</span>
              </div>
              <div className="contact-item">
                <FaMapMarkerAlt />
                <span>Sheikh Zayed Road, Dubai</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            className="footer-section"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
          >
            <h3>Компания</h3>
            <ul className="footer-links">
              {footerLinks.company.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.href}
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection(link.href);
                    }}
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </motion.div>

          <motion.div
            className="footer-section"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
          >
            <h3>Услуги</h3>
            <ul className="footer-links">
              {footerLinks.services.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.href}
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection(link.href);
                    }}
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </motion.div>

          <motion.div
            className="footer-section"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            viewport={{ once: true }}
          >
            <h3>Недвижимость</h3>
            <ul className="footer-links">
              {footerLinks.properties.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.href}
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection(link.href);
                    }}
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </motion.div>

          <motion.div
            className="footer-section footer-social"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            viewport={{ once: true }}
          >
            <h3>Социальные сети</h3>
            <p>Следите за нашими новостями и обновлениями</p>
            <div className="social-links">
              {socialLinks.map((social, index) => (
                <motion.a
                  key={index}
                  href={social.href}
                  className="social-link"
                  whileHover={{ scale: 1.1, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  title={social.name}
                >
                  <social.icon />
                </motion.a>
              ))}
            </div>
            
            <div className="newsletter">
              <h4>Подписка на новости</h4>
              <div className="newsletter-form">
                <input
                  type="email"
                  placeholder="Ваш email"
                  className="newsletter-input"
                />
                <motion.button
                  className="newsletter-btn"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Подписаться
                </motion.button>
              </div>
            </div>
          </motion.div>
        </div>

        <motion.div
          className="footer-bottom"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          viewport={{ once: true }}
        >
          <div className="footer-bottom-content">
            <p>&copy; 2024 Dubai Elite Properties. Все права защищены.</p>
            <div className="footer-bottom-links">
              <a href="#">Политика конфиденциальности</a>
              <a href="#">Условия использования</a>
              <a href="#">Карта сайта</a>
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  );
};

export default Footer;
