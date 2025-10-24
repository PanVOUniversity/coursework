import React from 'react';
import { motion } from 'framer-motion';
import { FaRocket, FaPhone, FaEnvelope, FaMapMarkerAlt, FaFacebook, FaInstagram, FaLinkedin } from 'react-icons/fa';
import './Footer.css';

const Footer = () => {
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
              <FaRocket className="logo-icon" />
              <span>Creative Agency</span>
            </div>
            <p className="footer-description">
              Ваш партнер в мире креативного маркетинга. 
              Более 5 лет опыта создания уникальных маркетинговых решений.
            </p>
            <div className="footer-contact">
              <div className="contact-item">
                <FaPhone />
                <span>+7 (495) 123-45-67</span>
              </div>
              <div className="contact-item">
                <FaEnvelope />
                <span>info@creativeagency.ru</span>
              </div>
              <div className="contact-item">
                <FaMapMarkerAlt />
                <span>Москва, ул. Арбат, 1</span>
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
            <h3>Услуги</h3>
            <ul className="footer-links">
              <li><a href="#services" onClick={(e) => { e.preventDefault(); scrollToSection('#services'); }}>Брендинг</a></li>
              <li><a href="#services" onClick={(e) => { e.preventDefault(); scrollToSection('#services'); }}>Реклама</a></li>
              <li><a href="#services" onClick={(e) => { e.preventDefault(); scrollToSection('#services'); }}>SMM</a></li>
              <li><a href="#services" onClick={(e) => { e.preventDefault(); scrollToSection('#services'); }}>SEO</a></li>
            </ul>
          </motion.div>

          <motion.div
            className="footer-section"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
          >
            <h3>Социальные сети</h3>
            <div className="social-links">
              <motion.a href="#" className="social-link" whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }}>
                <FaFacebook />
              </motion.a>
              <motion.a href="#" className="social-link" whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }}>
                <FaInstagram />
              </motion.a>
              <motion.a href="#" className="social-link" whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }}>
                <FaLinkedin />
              </motion.a>
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
            <p>&copy; 2024 Creative Agency. Все права защищены.</p>
          </div>
        </motion.div>
      </div>
    </footer>
  );
};

export default Footer;
